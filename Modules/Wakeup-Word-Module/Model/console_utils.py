from tqdm import tqdm


def write(txt):
    with open("logfile.txt", "a+") as f:
        f.write(str(txt))
        tqdm.write(str(txt))


# Generates a header in the console
def header(txt: object):
    write("+" + "-" * 100)
    write(
        f"| {txt.__str__().upper()}",
    )
    write("+" + "-" * 100)


# Generates a header in the console
def subheader(txt):
    write(f">>> {txt}")


# Generates a message in the console
def log(txt):
    write(f"    {txt}")


# Creates a directory and logs
def mkdir(path):
    subheader(path)
    path.mkdir(parents=True, exist_ok=True)
    log("Done")
    return path
