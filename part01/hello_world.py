from time import sleep

i = 0
while True:
    # Lower case base string
    hw = 'hello, world!'

    # Upper the ith index
    hw = hw[:i] + hw[i].upper() + hw[i + 1:]

    # Print the animated string, overwriting the previous line
    print('\r' + hw, end='')

    # Increment the index with modulo for looping
    i = (i + 1) % len(hw)

    # Arbitrary delay
    sleep(0.5)