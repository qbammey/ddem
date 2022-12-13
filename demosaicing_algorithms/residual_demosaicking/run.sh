#!/usr/bin/env bash
set -e


virtualenv=$1
bin=$2
pattern=$3
noise_sigma=$4
sigma=$5



# activate virtualenv
if [ ! -d $virtualenv ]; then
    echo "Virtualenv not found" > demo_failure.txt
    exit 0
fi
source $virtualenv/bin/activate


# run the experiments
for i in ARI HA GBTF RI MLRI WMLRI; do
    python ${bin}run.py --input input_0.png --pattern $pattern --Algorithm $i --output output_1_$i.png --output_diff output_2_$i.png --mosaic output_3.png --noise_sigma $noise_sigma --sigma $sigma > cpsnr_$i.txt &
done 

wait


# collect the results
echo "" > cpsnr_out.txt
for i in HA GBTF RI MLRI WMLRI ARI; do

	echo "---------- $i ----------" >> cpsnr_out.txt
	cat cpsnr_$i.txt >> cpsnr_out.txt
	echo "" >> cpsnr_out.txt
done
