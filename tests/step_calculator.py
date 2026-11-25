
a = [
    1,
    0,
    0,
    0,
    0,
    1,
]

i = 0

while i < len(a):
    
    if i == 0 or a[i] < 1:
        i += 1
    else:
        if a[i] > 0:
            
            if i-1 > 0:
                a[i] -= 1
                a[i-1] = a[0] + 1
                a[0] = 1
            else:
                a[i-1] = a[i] + 1
                a[i] = 0
                
            if i > 1:
                i -= 1
            
            if i > 1:
                print(f"\r{a}", end="", flush=True)

print("")      
print(a[0])
        
    