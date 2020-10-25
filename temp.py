from tqdm import tqdm

list1 = list(range(100,110))

for i,l in tqdm(enumerate(list1)):
    print(i,l)