import subprocess

def file_len(fname):
    p = subprocess.Popen('zcat ' + fname +' | wc -l', stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE, shell=True)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def flatten( aList ):
    return list(y for x in aList for y in x)
