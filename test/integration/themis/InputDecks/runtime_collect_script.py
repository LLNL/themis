import json


result = {
	"X": %%X%%,
	"Y": %%Y%%,
}

with open("output.json", "w") as file_handle:
	json.dump(result, file_handle)
