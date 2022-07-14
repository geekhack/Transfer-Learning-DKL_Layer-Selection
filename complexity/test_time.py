import timeit

start = timeit.default_timer()
a = 10+5
print("This is a good start", a)
print("This is a good start")
print("This is a good start")
print("This is a good start")


# All the program statements
stop = timeit.default_timer()
execution_time = stop - start

print("Program Executed in "+str(execution_time)) # It returns time in seconds