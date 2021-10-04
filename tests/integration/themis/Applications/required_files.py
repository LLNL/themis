import os
import json


def main():
    for expected_json_path in (
        os.path.join("required_dir", "required_json.json"),
        "required_json.json",
    ):
        with open(expected_json_path) as file_handle:
            info = json.load(file_handle)
        if info["foo"] != "bar":
            raise ValueError()


if __name__ == "__main__":
    main()
