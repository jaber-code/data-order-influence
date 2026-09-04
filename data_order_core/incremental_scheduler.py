

class IncrementalScheduler:
    def __init__(self, classes_step_size, normal_iters, total_classes):
        self.classes_step_size = classes_step_size
        self.normal_iters = normal_iters
        self.total_classes = total_classes
    
    def estimate_constant_iters(self):


        n = ((self.total_classes - self.classes_step_size) // self.classes_step_size) + 1
        
        classes = [self.classes_step_size + self.classes_step_size * i for i in range(n)]
        sum_classes = sum(classes)
        constant_iters = (self.normal_iters * self.total_classes) / sum_classes

        return 1, constant_iters
    
    def estimate_decay(self, iters_1, min_iters=2, max_iterations=1000, tolerance=500 ):

        n = ((self.total_classes - self.classes_step_size) // self.classes_step_size) + 1
        classes = [self.classes_step_size + self.classes_step_size * i for i in range(n)]

        target = self.normal_iters * self.total_classes
        target = target - target * 0.1
        def calculate_sum(decay_factor):
            total = 0
            current_iters = iters_1
            for i in range(n):
                if current_iters < min_iters:
                    current_iters = min_iters
                total += current_iters * classes[i]
                current_iters = current_iters * decay_factor
            #    if current_iters < 16:
            #        current_iters += ( current_iters*1.1)

            return total
        
        low, high = 0, 1
        target = target - target * 0.02
        for _ in range(max_iterations):
            mid = (low + high) / 2
            current_sum = calculate_sum(mid)
            print("current_sum: ", current_sum, "  mid: ", mid)
            if abs(current_sum - target) < tolerance:
                break
            elif current_sum > target:
                high = mid
            else:
                low = mid
        
        return mid, iters_1
    
    def estimate_scale(self, iters_1, max_iters=2, max_iterations=1000, tolerance=500 ):

        n = ((self.total_classes - self.classes_step_size) // self.classes_step_size) + 1
        classes = [self.classes_step_size + self.classes_step_size * i for i in range(n)]

        target = self.normal_iters * self.total_classes
        
        def calculate_sum(scale_factor):
            total = 0
            current_iters = iters_1
            for i in range(n):
                if current_iters > max_iters:
                    current_iters = max_iters
                total += current_iters * classes[n - i - 1]
                
                current_iters = current_iters * scale_factor
                print("current_iters: ", current_iters)

            return total
        
        low, high = 1, 2
        target = target - target * 0.3
        for _ in range(max_iterations):
            mid = (low + high) / 2
            current_sum = calculate_sum(mid)
            print("current_sum: ", current_sum, "  mid: ", mid)
            if abs(current_sum - (target)) < tolerance:
                break
            elif current_sum > target:
                high = mid
            else:
                low = mid
        
        return mid, iters_1

"""
if __name__ == '__main__':
    try:
        print("hi")
        inc = True

        step_size = 20
        noral_epochs = 100
        total_classes = 1000

        sc = IncrementalScheduler(classes_step_size=step_size, normal_iters=noral_epochs, total_classes=total_classes)

        if inc:
            end_iters = noral_epochs * 0.4
            iters_1 = 1

            scale_factor, start_iter  = sc.estimate_scale(iters_1=iters_1, max_iters=end_iters)
            print(scale_factor,"  ", start_iter)
            
            n = ((total_classes - step_size) // step_size) + 1
            classes = [step_size + step_size * i for i in range(n)]

            target = noral_epochs * total_classes
            
            total = 0
            current_iters = iters_1
            for i in range(len(classes)):
                if current_iters > end_iters:
                    current_iters = end_iters
                print(i, "  current_iters, ", current_iters)

                total += current_iters * classes[n - i - 1]
                current_iters = current_iters * scale_factor
            print(total, "  vs  target= " , target)
        else:
            end_iters = 1
            iters_1 = noral_epochs * 0.4
            decay_factor, start_iter  = sc.estimate_decay(iters_1=iters_1, min_iters=end_iters)
            print(decay_factor,"  ", start_iter)
            
            n = ((total_classes - step_size) // step_size) + 1
            classes = [step_size + step_size * i for i in range(n)]

            target = noral_epochs * total_classes
            
            total = 0
            current_iters = iters_1
            for i in range(n):
                if current_iters < end_iters:
                    current_iters = end_iters
                print(i, "  current_iters, ", current_iters)

                total += current_iters * classes[i]
                current_iters = current_iters * decay_factor
            print(total, "  vs  target= " , target)

    except Exception as e:
        print(f"Error: {e}")
"""