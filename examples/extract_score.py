import re
import sys
import csv

infile = sys.argv[1]
outfile = sys.argv[2]
results = dict()
with open(infile, 'r') as f:
    for line in f:
        line = line.strip()
        token = line.split("Running with lang = ")[1]
        lang = token.split("_")[0].upper()
        assert lang not in results

        line2 = f.readline().strip()
        metrics = re.findall("\d+\.\d+", line2)
        UAS, LAS = float(metrics[0]), float(metrics[1])
        results[lang] = {'uas': UAS, 'las': LAS}

avg_uas, avg_las = 0, 0
fw = open(outfile, 'w')
csv_writer = csv.writer(fw, delimiter=',')
for lang, vals in results.items():
    # print('%s\t%.2f\t%.2f' % (lang, vals['uas'], vals['las']))
    csv_writer.writerow([lang, '%.2f' % vals['uas'], '%.2f' % vals['las']])
    avg_uas += vals['uas']
    avg_las += vals['las']

avg_uas = avg_uas / len(results)
avg_las = avg_las / len(results)
csv_writer.writerow(['AVG', '%.2f' % avg_uas, '%.2f' % avg_las])
print('AVG\t%.2f\t%.2f' % (avg_uas, avg_las))
fw.close()
