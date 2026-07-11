use clap::Parser;
use pt_fuser::trace::metrics::MetricsRange;
use pt_fuser::trace::{Chunk, Frame, Trace};

/// Compare a baseline (fast) merged trace against a slow merged trace. Walks the aligned
/// call tree and, for every frame whose time grew, reports how much and whether that extra
/// time was an off-CPU stall, an on-CPU stall, or real work -- attributing it as deep in
/// the call tree as the data allows.
#[derive(Parser)]
struct Cli {
    /// baseline (fast) pt-fuser trace
    baseline: String,
    /// slow pt-fuser trace
    slow: String,
    #[clap(long, default_value_t = false, help = "input trace files are gzipped")]
    gzip: bool,
    #[clap(
        long,
        default_value_t = 1000,
        help = "only print frames whose |time delta| in ns is at least this"
    )]
    min_delta: u64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    OffCpu,
    OnCpu,
    Work,
    Faster,
    Same,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::OffCpu => "off-CPU stall (descheduled / interrupt)",
            Kind::OnCpu => "on-CPU stall (cache / memory / mispredict)",
            Kind::Work => "more work (extra instructions)",
            Kind::Faster => "faster in slow trace",
            Kind::Same => "no significant change",
        }
    }
}

/// Classify why `slow`'s time differs from `base` along the three metric axes. Assumes the
/// caller only invokes this when the time delta is significant; returns `Same` otherwise.
///   time up, no extra cycles, no extra insn -> off-CPU  (thread was not scheduled)
///   time up, extra cycles, no extra insn    -> on-CPU stall  (cycles burned, none retired)
///   time up, extra insn                     -> genuinely more work
///   time down                               -> faster in the slow trace
fn classify(base: &MetricsRange, slow: &MetricsRange, min_delta: u64) -> Kind {
    let dt = slow.total_time() as i64 - base.total_time() as i64;
    let dc = slow.total_cycles() as i64 - base.total_cycles() as i64;
    let di = slow.total_insn() as i64 - base.total_insn() as i64;
    if dt.unsigned_abs() < min_delta {
        Kind::Same
    } else if dt < 0 {
        Kind::Faster
    } else if di > 0 {
        Kind::Work
    } else if dc > 0 {
        Kind::OnCpu
    } else {
        Kind::OffCpu
    }
}

fn child_frames(frame: &Frame) -> Vec<&Frame> {
    frame
        .chunks()
        .iter()
        .filter_map(|c| match c {
            Chunk::Frame(f) => Some(f),
            _ => None,
        })
        .collect()
}

fn load(path: &str, gzip: bool) -> Trace {
    let data = std::fs::read(path).expect("failed to read trace file");
    Trace::bin_deserialize(&data, gzip).expect("trace file is malformed")
}

struct Row {
    depth: usize,
    symbol: String,
    delta_time: i64,
    delta_cycles: i64,
    delta_insn: i64,
    kind: Kind,
    note: &'static str,
}

/// Recursively align `base` and `slow` frame-by-frame. Children are paired by symbol name in
/// order of appearance. We always descend into every matched pair -- a frame that got slower
/// can be masked at its parent by a faster sibling, so pruning the walk by the parent's net
/// delta would risk hiding a real hotspot. `min_delta` therefore controls only which frames
/// are printed, never which are explored. Rows carry their true call depth, so if a quiet
/// intermediate frame is skipped its indentation level is simply absent. Frames present on
/// only one side are reported as structural divergences.
fn diff_frames(base: &Frame, slow: &Frame, depth: usize, min_delta: u64, out: &mut Vec<Row>) {
    let base_children = child_frames(base);
    let slow_children = child_frames(slow);
    let mut used = vec![false; slow_children.len()];

    for &b in &base_children {
        let mut matched: Option<&Frame> = None;
        for (i, &s) in slow_children.iter().enumerate() {
            if !used[i] && s.symbol.name == b.symbol.name {
                used[i] = true;
                matched = Some(s);
                break;
            }
        }

        match matched {
            Some(s) => {
                let dt = s.metrics.total_time() as i64 - b.metrics.total_time() as i64;
                if dt.unsigned_abs() >= min_delta {
                    out.push(Row {
                        depth,
                        symbol: b.symbol.name.clone(),
                        delta_time: dt,
                        delta_cycles: s.metrics.total_cycles() as i64
                            - b.metrics.total_cycles() as i64,
                        delta_insn: s.metrics.total_insn() as i64 - b.metrics.total_insn() as i64,
                        kind: classify(&b.metrics, &s.metrics, min_delta),
                        note: "",
                    });
                }
                // Always descend, even if this frame looked quiet: the extra time may live in
                // a child whose slowdown was cancelled out here by a faster sibling.
                diff_frames(b, s, depth + 1, min_delta, out);
            }
            None => out.push(Row {
                depth,
                symbol: b.symbol.name.clone(),
                delta_time: -(b.metrics.total_time() as i64),
                delta_cycles: -(b.metrics.total_cycles() as i64),
                delta_insn: -(b.metrics.total_insn() as i64),
                kind: Kind::Same,
                note: "only in baseline",
            }),
        }
    }

    for (i, &s) in slow_children.iter().enumerate() {
        if !used[i] {
            out.push(Row {
                depth,
                symbol: s.symbol.name.clone(),
                delta_time: s.metrics.total_time() as i64,
                delta_cycles: s.metrics.total_cycles() as i64,
                delta_insn: s.metrics.total_insn() as i64,
                kind: Kind::Same,
                note: "only in slow",
            });
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let base = load(&cli.baseline, cli.gzip);
    let slow = load(&cli.slow, cli.gzip);

    let base_root = base.root_frame();
    let slow_root = slow.root_frame();

    let root_dt = slow_root.metrics.total_time() as i64 - base_root.metrics.total_time() as i64;
    println!("Comparing `{}`", base_root.symbol.name);
    println!(
        "  baseline: {} ns, slow: {} ns, delta: {:+} ns\n",
        base_root.metrics.total_time(),
        slow_root.metrics.total_time(),
        root_dt
    );

    let mut rows = Vec::new();
    diff_frames(base_root, slow_root, 0, cli.min_delta, &mut rows);

    if rows.is_empty() {
        println!("No divergence above {} ns.", cli.min_delta);
        return;
    }

    println!(
        "{:>12} {:>12} {:>12}   {}",
        "d_time(ns)", "d_cycles", "d_insn", "frame  (indented by call depth)"
    );
    for r in &rows {
        let indent = "  ".repeat(r.depth);
        let label = if r.note.is_empty() { r.kind.label() } else { r.note };
        println!(
            "{:>+12} {:>+12} {:>+12}   {}{}  [{}]",
            r.delta_time, r.delta_cycles, r.delta_insn, indent, r.symbol, label
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pt_fuser::trace::metrics::Metrics;

    fn range(ts: u64, cycles: u64, insn: u64) -> MetricsRange {
        MetricsRange::new(Metrics::constant(0), Metrics::new(ts, cycles, insn))
    }

    #[test]
    fn off_cpu_when_time_up_but_no_extra_work() {
        assert_eq!(
            classify(&range(100, 100, 100), &range(300, 100, 100), 10),
            Kind::OffCpu
        );
    }

    #[test]
    fn on_cpu_when_cycles_up_but_insn_flat() {
        assert_eq!(
            classify(&range(100, 100, 100), &range(300, 400, 100), 10),
            Kind::OnCpu
        );
    }

    #[test]
    fn work_when_more_instructions() {
        assert_eq!(
            classify(&range(100, 100, 100), &range(300, 400, 250), 10),
            Kind::Work
        );
    }

    #[test]
    fn same_when_below_threshold() {
        assert_eq!(
            classify(&range(100, 100, 100), &range(105, 100, 100), 10),
            Kind::Same
        );
    }

    #[test]
    fn faster_when_slow_is_quicker() {
        assert_eq!(
            classify(&range(300, 300, 300), &range(100, 100, 100), 10),
            Kind::Faster
        );
    }
}
