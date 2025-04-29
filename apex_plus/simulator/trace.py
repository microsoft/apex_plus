from dataclasses import dataclass
from typing import List
import json
import csv
from datetime import datetime


# Function to parse the CSV and calculate offsets
def parse_csv(file_path):
    offsets = []
    with open(file_path, mode="r") as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip the header row if there is one
        first_row = next(csv_reader)
        # NOTE the timestamp in the example csv file has seven digits after the decimal point
        # yet, datetime.strptime() only supports six digits after the decimal point
        # so a [:-1] in first_row[0][:-1] is used to trim the last digit
        # example csv: https://github.com/Azure/AzurePublicDataset/blob/master/data/AzureLLMInferenceTrace_conv.csv
        zero_time = datetime.strptime(
            first_row[0][:-1], "%Y-%m-%d %H:%M:%S.%f"
        )  # Adjust the format as needed
        offsets.append([0, first_row[1], first_row[2]])  # First row offset is zero

        for row in csv_reader:
            current_time = datetime.strptime(
                row[0][:-1], "%Y-%m-%d %H:%M:%S.%f"
            )  # Adjust the format as needed
            offset = (current_time - zero_time).total_seconds()
            offsets.append([offset, row[1], row[2]])

    return offsets


@dataclass
class Request:

    input_len: int
    output_len: int
    time_stamp: int  # offset to the first request in nanosec.


class Trace:

    def __init__(
        self,
        requests: List[Request],
    ) -> None:
        self.requests = requests

    @classmethod
    def from_static(
        cls,
        num_requests: int,
        input_len: int,
        output_len: int,
    ) -> "Trace":
        requests = [Request(input_len, output_len, 0) for _ in range(num_requests)]
        return cls(requests)

    @classmethod
    def from_dynamic(
        cls,
        file_path: str,
    ) -> "Trace":
        requests = []
        if file_path.endswith(".jsonl"):
            with open(file_path, "r") as file:
                for line in file:
                    data = json.loads(line)
                    requests.append(
                        Request(
                            data.get("ContextTokens"),
                            data.get("GeneratedTokens"),
                            int(1 * (data.get("StartTimeOffset")) // 1e3),
                        )
                    )  # ns -> us
        elif file_path.endswith(".csv"):
            offsets = parse_csv(file_path)
            for req in offsets:
                requests.append(Request(int(req[1]), int(req[2]), int(req[0] * 1e6)))
        else:
            print("Unsupported file format")

        return cls(requests)
