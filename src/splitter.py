import argparse
import ast

def split_file_to_list(file_path, separator=None):
    """Reads file and returns a list of values using either a provided or auto-detected separator."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if separator:
        return [item.strip() for item in content.split(separator) if item.strip()]

    # Try newline first
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    # Try common inline separators
    candidates = [",", ";", "|", "\t"]
    best_sep = max(candidates, key=lambda sep: content.count(sep))
    return [item.strip() for item in content.split(best_sep) if item.strip()]

def format_list(items, style):
    """Formats a list of strings according to the selected style."""
    if style == "newline":
        return "\n".join(items)
    elif style == "quoted-newline-comma":
        return ",\n".join(f"'{item}'" for item in items)
    elif style == "comma":
        return ",".join(items)
    elif style == "quoted-comma":
        return ",".join(f"'{item}'" for item in items)
    else:
        raise ValueError("Unknown format style.")

def main():
    parser = argparse.ArgumentParser(description="Split or format lists from file or string input.")
    parser.add_argument("-m", "--mode", choices=["split", "format"], required=True,
                        help="Choose 'split' to split file contents, or 'format' to format a list.")
    parser.add_argument("-f", "--file", help="Path to input file.")
    parser.add_argument("-s", "--separator", help="Separator to use when splitting file contents. If omitted, auto-detect.")
    parser.add_argument("-l", "--list", help="Inline Python-style list string (e.g. \"['a','b','c']\")")
    parser.add_argument("-F", "--format", choices=["newline", "quoted-newline-comma", "comma", "quoted-comma"],
                        help="Output format style for list printing.")

    args = parser.parse_args()

    # Ensure logging is configured for CLI scripts (INFO -> stdout)
    from src.utils import setup_logging, get_logger
    setup_logging(verbosity=1)
    logger = get_logger()

    if args.mode == "split":
        if not args.file:
            logger.error("Error: --file is required in split mode.")
            return
        result = split_file_to_list(args.file, args.separator)
        logger.info(result)

    elif args.mode == "format":
        if args.list:
            try:
                items = ast.literal_eval(args.list)
                if not isinstance(items, list):
                    raise ValueError
            except Exception:
                logger.error("Error: --list must be a valid Python list string.")
                return
        elif args.file:
            items = split_file_to_list(args.file)
        else:
            logger.error("Error: Either --list or --file must be provided in format mode.")
            return

        try:
            formatted = format_list(items, args.format)
            logger.info(formatted)
        except ValueError as e:
            logger.error("Error: %s", e)

if __name__ == "__main__":
    main()
