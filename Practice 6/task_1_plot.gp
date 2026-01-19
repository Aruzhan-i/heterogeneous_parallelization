set datafile separator ','
set terminal pngcairo size 600,400
set output 'task_1_performance.png'
set title 'OpenCL vector_add: CPU vs GPU'
set ylabel 'Average kernel time (ms)'
set yrange [0:*]
set grid ytics
set style data histograms
set style fill solid 0.6
set boxwidth 0.5
unset key
plot 'task_1_results.csv' using 2:xtic(1) every ::1 title 'Execution time'
