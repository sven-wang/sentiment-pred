#!/usr/bin/env bash
for i in 0.05 0.2
do
   # your-unix-command-here

   echo $i
   date
   python2 part5Perceptron.py $i 0
   python evalResult.py EN/dev.out EN/dev.perceptron.out
done
date

