
def black_adam(grad_funs, init_params, callback=None, num_iters=100,
               step_size=0.001, b1=0.9, b2=0.999, eps=10**-8, num_proc = 1):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    multi_grad = []
    for proc_num in range(num_proc):
        flattened_grad, _, _ = flatten_func(grad_funs[proc_num], init_params)
        multi_grad.append(flattened_grad)
    x, unflatten = flatten(init_params)
    multi_grad = tuple(multi_grad)
    manage = mp.Manager()
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    result_queue = manage.Queue()
    pool = mp.Pool(num_proc)
    print num_iters
    for i in range(num_iters):
        print "here we are"
        for proc in range(0,num_proc) :
            pool.apply_async(tricky_wrapper, args=(multi_grad[proc],x,proc,result_queue))
        print "pool please"
        print "and here"
        results = []
        for answer in range(num_proc):
            print "t"
            results.append(result_queue.get())
        g = np.array(reduce(lambda a,b : a+b,results,np.array(0)))

        if callback: callback(unflatten(x), i, unflatten(g))

        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        print multi_grad
        #multi_grad = [multi_grad[2]]*4
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return unflatten(x)


def gen_grads(data,num_proc):
    num_rows, num_col = data.shape
    gradfuns = []
    ranges = disseminate_values(num_rows,num_proc)
    prev_pos = 0
    for x in range(num_proc) :
        user_range = range(prev_pos,prev_pos+ranges[x])
        prev_pos += ranges[x]
        gradfuns.append(lossGradMultiCore(data,user_range,num_proc))

    return tuple(gradfuns)

def tricky_wrapper(grad_fun,params,iter,queue):
    print "we're in ", iter
    queue.put(grad_fun(params,iter))
    print "we're out ", iter

def lossGradMultiCore(data,user_range, num_proc = 1):
    return grad(lambda params,_:nnLoss(params,data=data,indices=user_range, num_proc = num_proc))

def dataCallback(data):
    return lambda params,iter,grad: print_perf(params,iter,grad,data=data)


def disseminate_values(num_items,num_bins):
    chunk_size = num_items/num_bins
    ranges = [chunk_size]*num_bins
    remainder = num_items-num_bins*chunk_size
    #Remainder < num_bins
    for x in range(remainder):
        ranges[x] += 1
    return ranges