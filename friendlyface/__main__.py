"""Run FriendlyFace server: python3 -m friendlyface"""

import uvicorn


def main() -> None:
    uvicorn.run("friendlyface.api.app:app", host="0.0.0.0", port=3849, reload=True)


if __name__ == "__main__":
    main()
