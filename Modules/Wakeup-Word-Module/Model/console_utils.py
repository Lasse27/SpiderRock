# Generates a header in the console
def header(txt: object):
    print("+" + "-" * 100)
    print("|", txt.__str__().upper())
    print("+" + "-" * 100)


# Generates a header in the console
def subheader(*args):
    print(">>>", *args)


# Generates a message in the console
def log(*args):
    print("   ", *args)


# Creates a directory and logs
def mkdir(path):
    subheader(path)
    path.mkdir(parents=True, exist_ok=True)
    log("Done")
    return path
