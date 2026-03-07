from datasets import load_dataset
ds = load_dataset("bowen-upenn/PersonaMem-v2")
print(ds)
print(ds[list(ds.keys())[0]][0].keys())
print(ds[list(ds.keys())[0]][0])
