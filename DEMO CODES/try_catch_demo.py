

try:
    age = int(input("Age: "))
    print(age)
except Exception as ex:
    print("Invalid age : please enter valid age...")
    print(type(ex))
else:
    print("not exception was thrown")
finally:
    print("run if exception or not")


 

