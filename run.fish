#!/usr/bin/env fish

set location (echo $argv[1] | sed 's,/$,,')

for mov in (ls $location/*.{mov,MOV})
    ./spot_larva.py $mov -wh 750 -f 0.7 | tee {$mov}_result.txt
end
