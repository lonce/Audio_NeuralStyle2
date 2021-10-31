#!/bin/bash
for i in *.mp3
do
    sox "$i" "$(basename -s .mp3 "$i").wav" remix 1-2
done
