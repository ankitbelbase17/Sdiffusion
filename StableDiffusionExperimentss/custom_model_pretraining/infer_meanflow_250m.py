from infer_variant import build_parser, run_inference


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.approach = "meanflow"
    args.model_size = "250m"
    run_inference(args)

