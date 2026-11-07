"""Guard historical sentinel-based commands after the official target revision."""


def add_legacy_target_flag(parser):
    parser.add_argument("--allow-legacy-target", action="store_true",
                        help="Historical replay only: old 10000 sentinel targets are obsolete")


def check_legacy_target(parser, args):
    if not args.allow_legacy_target:
        parser.error("Obsolete 10000-sentinel target protocol. Import official train_v2.csv "
                     "and use scripts/submit_v2.py. For historical replay only, "
                     "pass --allow-legacy-target.")
