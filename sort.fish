#!/usr/bin/env fish -d 1

set src (echo $argv[1] | sed 's,/$,,')

set dst $src/(date '+%Y-%m-%d')_result
set text $dst/all_text
set images $dst/all_images
set groups $dst/grouped_results

test -d $dst; or mkdir -v $dst
test -d $text; or mkdir -v $text
test -d $images; or mkdir -v $images
test -d $groups; or mkdir -v $groups

# move logs & results to result-text dir
mv $src/*_result.{csv,txt} $text
# produce whole-result overall tables, move to result-top dir
./combine_csvs.py -y $text/*_result.csv
mv $text/overall*.csv $dst
# produce whole-result args, pathcount, and err/warn summaries in result-top dir
grep 'args' $text/*_result.csv > $dst/meta_args.txt
grep '=.*paths' $text/*_result.txt > $dst/meta_paths.txt
grep 'Error\|Warning' $text/*_result.txt > $dst/meta_problems.txt

# move images to result-image dir, and produce overviews in result-top dir
mv $src/*_result*.png $images
for f in (ls $images/*circles.png)
    echo "<p>$f<br/><img src=\"$f\"></p>"
end > $dst/meta_circles.html
for f in (ls $images/*T3.0.png)
    set fc (echo $f | sed 's/T3.0/colors/')
    echo "<p style=\"white-space:nowrap;\">$f<br/><img src=\"$f\"><img src=\"$fc\" style=\"zoom:2;\"></p>"
end > $dst/meta_T3.html

set groupnames (cat $src/meta_group*.txt | python2 -c '''
import sys,re
for line in (ln.strip() for ln in sys.stdin):
    folder = re.sub(r"[^-\w,]", "", line.replace(":","-"))
    print folder
''')
echo === Identified groups $groupnames

for grp in $groupnames
    set group $groups/$grp
    test -d $group; or mkdir -v $group

    set glob (echo $grp | python2 -c '''
import sys
nums = map(int, sys.stdin.read().split("-", 1)[1].split(","))
print "*{%s}_result.csv" % ",".join(map(str, nums))
''')
    echo == Globbing for group $grp
    set csvs (eval "ls $text/$glob") # Is there a better way to expand a glob stored in a string?
    if test (count $csvs) -gt 0
        cp $csvs $group
    end
    ./combine_csvs.py -y $group/*_result.csv
end

