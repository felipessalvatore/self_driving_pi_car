#!/bin/bash
PARENT_DIR=$(pwd)
for d in $(find $1 -maxdepth 1 -type d)
do
	FC="$(ls -1 $d | grep -E png -c)"
	if [ $FC = 0 ]
		then
			continue
	fi
	FC=$(($FC-1))
	cd $d
	echo "Converting images from $d to gif..."
	CMD="convert -loop 0 {0..$FC}.png run.gif"
	eval "$CMD"
	cd $PARENT_DIR
done
echo 'All done!'