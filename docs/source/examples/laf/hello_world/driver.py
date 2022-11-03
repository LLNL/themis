from themis import laf
from trata.composite_samples import parse_file

samples = parse_file("my_samples.csv", "csv")
mgr = laf.BatchSubmitter("script.sh", samples, None)
print(mgr.execute())
