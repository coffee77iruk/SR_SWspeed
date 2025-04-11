import os
import argparse
from datetime import datetime
from sunpy.time import parse_time
from A_CH_processor import process_a_ch  # A_CH_processor.py

def main():
    parser = argparse.ArgumentParser(
        description="FITS 파일들의 A_CH 값을 시간별로 추출하여 txt 파일로 저장합니다."
    )
    parser.add_argument("--channel", type=str, required=True,
                        help="채널 폴더 이름 (예: '193' 또는 '193,211')")
    parser.add_argument("--start", type=str, required=True,
                        help="시작 시간 (예: 2012-01-01)")
    parser.add_argument("--end", type=str, required=True,
                        help="종료 시간 (예: 2012-12-31)")
    parser.add_argument("--cadence", type=int, default=24,
                        help="출력 간격 (시간 단위, 예: 24 또는 12)")
    parser.add_argument("--base_dir", type=str, default=r"E:\Research\SR\input\CH",
                        help="FITS 파일들이 있는 최상위 폴더")
    parser.add_argument("--output", type=str, default="A_CH_results_{channel}.txt",
                        help="출력 txt 파일명. 여러 채널일 경우 '{channel}'라는 플레이스홀더를 사용할 수 있습니다.")
    args = parser.parse_args()

    start_dt = parse_time(args.start).to_datetime()
    end_dt = parse_time(args.end).to_datetime()
    
    channels = [chan.strip() for chan in args.channel.split(',')]
    for chan in channels:
        results = process_a_ch(chan, start_dt, end_dt, args.cadence, args.base_dir)
        
        # 출력 파일 이름 결정 (여러 채널일 경우 '{channel}' 플레이스홀더 사용)
        base_output = args.output
        if len(channels) > 1:
            if '{channel}' in base_output:
                output_file = base_output.format(channel=chan)
            else:
                filename, ext = os.path.splitext(base_output)
                output_file = f"{filename}_{chan}{ext}"
        else:
            output_file = base_output

        with open(output_file, "w") as f_out:
            for line in results:
                f_out.write(line + "\n")
        print(f"채널 {chan} 처리 완료. 결과는 {output_file}에 저장되었습니다.")


if __name__ == '__main__':
    main()


# To run this script, you can use the command line as follows:
# activate venv
# cd Research\SR\input\convert
# python A_CH_run.py --channel "193" --start "2012-01-01" --end "2012-12-31" --cadence 24 --base_dir "E:\Research\SR\input\CH" --output "A_CH_results_{channel}.txt"
# python A_CH_run.py --channel "193,211" --start "2012-01-01" --end "2012-12-31" --cadence 24 --base_dir "E:\Research\SR\input\CH" --output "A_CH_results_{channel}.txt"
