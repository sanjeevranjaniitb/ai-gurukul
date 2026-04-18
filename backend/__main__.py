"""Allow ``python -m backend.eval`` to work as a package entry point."""

from backend.eval import main

if __name__ == "__main__":
    main()
