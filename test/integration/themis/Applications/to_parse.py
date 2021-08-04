import required_script

x = %%X%%
y = %%Y%%

with open("my_script_out.csv", "w") as file_handle:
    file_handle.write("1,2,3,4\n")
    file_handle.write("5,6,7,8\n")
    file_handle.write(required_script.func())
    file_handle.write("{},{}\n".format(x, y))
