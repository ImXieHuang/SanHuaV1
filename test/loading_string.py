import time

loading = ["⠋", "⠙", "⠸", "⢰", "⣠", "⣄", "⡆", "⠇"]
i=0
while True:
    i = (i+1)%8
    print(f"\rLoading... {loading[i]}", end="")
    time.sleep(0.1)