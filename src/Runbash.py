# 环境：macosx、stackless python & twisted

#cmd = 'e:/git/enas/run_cnn_macro.sh'  # 全路径或者./相对路径
#cmd = 'e:/git/enas/run_rnn.sh'  # 全路径或者./相对路径
#cmd = 'e:/git/enas/fixed_cnn_macro.sh'  # 全路径或者./相对路径
#cmd = 'e:/git/enas/fixed_cnn_micro.sh'
cmd = 'd:/wjwang/code/enas/fixed_cnn_micro.sh'

import subprocess

p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
while p.poll() == None:
    line = p.stdout.readline()
    print(line)  # 必须执行print，否则一直不返回，原因不明
    result = result + line