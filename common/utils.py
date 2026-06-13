def print_title(title, width=70):
    print("-"*width)
    print(f"{'-'*5}{title:^{width-10}}{'-'*5}")
    print("-"*width)

def pretty_format(label, value, width=35, value_width=35):
    return f"{str(label + ':'):<{width}}{str(value):>{value_width}}"

def pretty_print(label, value, width=35, value_width=35):
    print(pretty_format(label, value, width, value_width))

