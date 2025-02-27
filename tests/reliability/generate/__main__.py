import argparse

from tests.reliability.generate import generate_test_cases

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate test cases by specifying configuration and input instructions."
    )
    parser.add_argument(
        "-d", "--dst_path", type=str, required=True, help="Destination path where generated test cases will be saved."
    )
    parser.add_argument(
        "-n", "--num_inputs", type=int, default=1, help="Number of input cases to generate (default: 1)."
    )
    parser.add_argument(
        "-p", "--program_instructions", type=str, help="Additional instructions for the generated test program."
    )
    parser.add_argument(
        "-i", "--input_instructions", type=str, help="Additional instructions for generating test inputs."
    )

    args = parser.parse_args()

    generate_test_cases(
        dst_path=args.dst_path,
        num_inputs=args.num_inputs,
        program_instructions=args.program_instructions,
        input_instructions=args.input_instructions,
    )
