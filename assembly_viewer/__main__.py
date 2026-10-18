"""Unified interface and explicit worker/export commands."""
import argparse
from datetime import datetime
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="单画布拓扑读取、实时组装和全过程回放")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("ui", "启动完整工作界面"), ("serve", "查看已有过程记录")):
        cmd = commands.add_parser(name, help=help_text)
        cmd.add_argument("--host", default="127.0.0.1")
        cmd.add_argument("--port", type=int, default=8765)
        if name == "ui":
            cmd.add_argument("--project")
        else:
            cmd.add_argument("trace_dir")
    for name in ("prepare", "run"):
        cmd = commands.add_parser(name, help="读取输入" if name == "prepare" else "从已读取资源启动组装")
        cmd.add_argument("--spec", required=True)
    record = commands.add_parser("record", help="批处理装配并记录真实过程")
    for name in ("nodes", "topology", "output"):
        record.add_argument("--" + name, required=True)
    record.add_argument("--config")
    demo = commands.add_parser("demo", help="生成明确标记的合成演示记录")
    demo.add_argument("--output", default=f"assembly_viewer/runs/demo_{datetime.now():%Y%m%d_%H%M%S}")
    export = commands.add_parser("export", help="离线回放、阶段图片和过程视频")
    export.add_argument("trace_dir")
    export.add_argument("--html")
    export.add_argument("--frames")
    export.add_argument("--video")
    export.add_argument("--scope", choices=("formal", "live", "forward"), default="formal", help="视频过程范围")
    export.add_argument("--all-connections", action="store_true")
    export.add_argument("--all-components", action="store_true")
    export.add_argument("--detailed", action="store_true")
    figure = commands.add_parser("figure", help="从已有分子 JSON 导出矢量图，无需重跑装配")
    figure.add_argument("molecule_json")
    figure.add_argument("--output-dir")
    figure.add_argument("--formats", nargs="+", choices=("svg", "pdf", "png", "tif", "tiff"),
                        default=("svg", "pdf", "png"))
    figure.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    if args.command in {"ui", "serve"}:
        from .server import serve
        return serve(project=getattr(args, "project", None), trace_dir=getattr(args, "trace_dir", None),
                     host=args.host, port=args.port)
    if args.command in {"prepare", "run"}:
        from . import worker
        return getattr(worker, args.command)(args.spec)
    if args.command == "record":
        from RL_MTCS import AssemblyConfig
        from .adapter import assemble_with_trace
        values = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
        result = assemble_with_trace(args.nodes, args.topology, output_dir=args.output, config=AssemblyConfig(**values))
        print(json.dumps({key: result[key] for key in ("status", "legal", "all_complete", "completion_report", "process_trace")}, ensure_ascii=False, indent=2))
    elif args.command == "demo":
        from RL_MTCS import AssemblyConfig
        from .adapter import record_state
        from .demo import demo_state
        from .export import export_html
        result = record_state(demo_state(), output_dir=args.output, synthetic=True,
                              config=AssemblyConfig(connection_candidate_limit=2, forward_max_rollbacks=1,
                                                    seed=17, verbose=False, save_stage_images=False))
        print(f"合成示例: {result['status']} · 合法: {result['legal']} · {result['process_trace']}")
        print(export_html(result["process_trace"], Path(args.output) / "replay.html"))
    elif args.command == "figure":
        from .export import export_molecule
        print(json.dumps(export_molecule(args.molecule_json, args.output_dir,
                         formats=args.formats, dpi=args.dpi), ensure_ascii=False, indent=2))
    elif args.command == "export":
        from .export import export_html, export_frames, export_video
        if not any((args.html, args.frames, args.video)):
            parser.error("请指定 --html、--frames 或 --video")
        if args.html:
            print(export_html(args.trace_dir, args.html, detailed=args.detailed))
        if args.frames:
            print(json.dumps(export_frames(args.trace_dir, args.frames,
                             kind="connections" if args.all_connections else "stages",
                             main_only=not args.all_components), ensure_ascii=False))
        if args.video:
            print(export_video(args.trace_dir, args.video, scope=args.scope, main_only=not args.all_components))


if __name__ == "__main__":
    main()
