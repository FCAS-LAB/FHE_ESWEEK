import argparse
from src.bilevel import bilevel
from src.Ablation.mapping_baseline.mapping_ablation import mapping_ablation
from src.Ablation.topology_baseline.topology_ablation import topology_ablation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_file', required=True)
    parser.add_argument('--bench_file', required=True)
    parser.add_argument('--trace_file', required=True)
    parser.add_argument('--topo_file', required=True)
    parser.add_argument('--show_file', required=True)
    args = parser.parse_args()

    # print("Time File:", args.time_file)
    # print("Benchmark File:", args.bench_file)
    # print("Trace File:", args.trace_file)
    # print("Topology File:", args.topo_file)
    # print("Show File:", args.show_file)

    print("Bilevel Algorithm:")
    bilevel(args.time_file, args.bench_file, args.trace_file, args.topo_file, args.show_file)
    print("-----------------------------------------------------")
    print("Mapping Ablation:")
    mapping_ablation(args.time_file, args.bench_file)
    print("-----------------------------------------------------")
    print("Topology Ablation:")
    topology_ablation(args.time_file, args.bench_file)

if __name__ == "__main__":
    main()