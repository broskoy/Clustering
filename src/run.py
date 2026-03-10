from compress import compress


def main():
    input_path = "input/windows.png"
    output_path = "output/windows.png"
    k = 64
    epsilon = 0.5
    inner_sample = 1000
    outer_sample = 1000

    compress(input_path, output_path, k, epsilon, inner_sample, outer_sample)


if __name__ == "__main__":
    main()