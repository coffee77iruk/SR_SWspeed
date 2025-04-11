import os
import argparse
from datetime import datetime
from sunpy.time import parse_time
from processor_A_ch import process_a_ch  # processor_A_ch.py

def main():
    parser = argparse.ArgumentParser(
        description="Extract the A_ch values of the FITS files over time and save them as txt files."
    )
    parser.add_argument("--channel", type=str, required=True,
                        help="channel name (e.g, '193' or '193,211')")
    parser.add_argument("--start", type=str, required=True,
                        help="start date (e.g, 2012-01-01)")
    parser.add_argument("--end", type=str, required=True,
                        help="end date (e.g, 2012-12-31)")
    parser.add_argument("--cadence", type=int, default=12,
                        help="cadence (e.g, 12 or 24 [hours])")
    parser.add_argument("--base_dir", type=str, default=r"E:\Research\SR\input\CH",
                        help="Parent folder with FITS files (e.g., E:\Research\SR\input\CH)")
    parser.add_argument("--output", type=str, default="A_CH_results_{channel}.txt",
                        help="textfile name. Use {channel} as a placeholder for the channel name.")
    args = parser.parse_args()

    start_dt = parse_time(args.start).to_datetime()
    end_dt = parse_time(args.end).to_datetime()
    
    channels = [chan.strip() for chan in args.channel.split(',')]   # e.g., [193,211]
    for chan in channels:
        results = process_a_ch(chan, start_dt, end_dt, args.cadence, args.base_dir)

        # output file name
        base_output = args.output
        if '{channel}' in base_output:
            output_file = base_output.format(channel=chan)
        else:
            filename, ext = os.path.splitext(base_output)
            output_file = f"{filename}_{chan}{ext}"

        # write a A_ch to output_file
        with open(output_file, "w") as f_out:
            for line in results:
                f_out.write(line + "\n")
        print(f"")
        print(f"complete the channel {chan}. The results are saved at {output_file}.")

if __name__ == '__main__':
    main()


# To run this script, you can use the command line as follows:
# activate venv
# cd Research\SR\input\convert
# python run_A_ch.py --channel "193" --start "2012-01-01" --end "2012-12-31" --cadence 24 --base_dir "E:\Research\SR\input\CH" --output "A_CH_results_{channel}.txt"
# python run_A_ch.py --channel "193,211" --start "2012-01-01" --end "2012-12-31" --cadence 24 --base_dir "E:\Research\SR\input\CH" --output "A_CH_results_{channel}.txt"
