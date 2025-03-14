from collections import defaultdict
import importlib
from riscv_application_profiler.consts import *
import riscv_application_profiler.consts as consts
from riscv_isac.log import *
from riscv_isac.plugins.spike import *
from riscv_application_profiler.plugins import instr_groups
from riscv_application_profiler import verif
from riscv_application_profiler import plugins
import riscv_config.isa_validator as isaval
from riscv_application_profiler.utils import Utilities
import os
import yaml
import sys
import multiprocessing as mp

def run(log, isa, output, verbose, config, cycle_accurate_config): #, check): 
    from rvop_decoder.rvopcodesdecoder import disassembler
    workers = mp.cpu_count()
    master_inst_dict = defaultdict(int)
    local_inst_dict = defaultdict(int)        
    arch = 'rv32'        
    
    def iterator(start,stop):
        spike_parser = spike()
        spike_parser.setup(trace=str(log), arch=arch)
        iterator = spike_parser.__iter__()
        curr = 0
        for line in iterator:
            curr +=1
            if curr < start:
                continue
            if curr > stop:
                break
            yield line
            
    def inst_counter(log):
        with open(log, 'r') as logfile:
            count = 0
            for _ in logfile:
                count += 1
            return count
        
    def decode(chunk):
        isac_decoder = disassembler()
        isac_decoder.setup(arch=arch)
        local_dict = defaultdict(int)
        for entry in chunk:
            decoded_instr = isac_decoder.decode(entry)
            key = sys.intern(decoded_instr.instr_name)
            if key in local_dict:
                local_dict[str(key)] += 1
            else:
                local_dict[str(key)] = 1
        master_queue.put(local_dict)
        
    insts = inst_counter(log)
    logger.info(f'No of Instructions Present: {insts}')
    logger.info(f'No of Workers: {workers}') 
    chunk = int(insts / workers)
    rem = int(insts % workers)
    w_chunk = [x * chunk for x in range(workers)] + [ ((workers * chunk) + rem)]
    for i in w_chunk:
        logger.info(f'Worker_chunk segments: {i}')
    w_proc = []
    master_queue = mp.Queue()
    
    for i in range(workers):
        decoded = mp.Process(target=decode,args=(iterator(w_chunk[i], w_chunk[i+1]),))
        decoded.start()
        w_proc.append(decoded)   
               
    for _ in range(len(w_proc)):
        local_inst_dict = master_queue.get() 
        for key, count in local_inst_dict.items():
            if key in master_inst_dict:
                master_inst_dict[str(key)] += count
            else:
                master_inst_dict[str(key)] = count
    for x in w_proc:
        x.join()

    logger.info(f'Total count of instructions parsed: {insts}')
    logger.info("Decoding...")
    logger.info("Done decoding instructions.")
    logger.info("Starting to profile...")
    
    utils = Utilities(log, output)
    utils.metadata()

    # Grouping by operations
    groups = [
        'loads',
        'stores',
        'imm computes',
        'imm shifts',
        'reg computes',
        'reg shifts',
        'jumps',
        'branches',
        "compares",
        "conversions",
        "moves",
        "classifies",
        "csrs",
        "fence",
    ]

    (extension_list, err, err_list) = isaval.get_extension_list(isa)

    for e in err_list:
        logger.error(e)
    if err:
        raise SystemExit(1)
    logger.info("Extensions Present in the log")
    for i in extension_list:
        logger.info(i)

    isa_arg = isa.split('I')[0]

    ret_dict, ext_ret_dict = instr_groups.group_by_operation(groups, isa_arg, extension_list, master_inst_dict, config, cycle_accurate_config)
    
    if 'C' in extension_list:
        logger.warning("riscv-isac does not decode immediate fields for compressed instructions. \
Value based metrics on branch ops may be inaccurate.")

    
    utils.tabulate_stats(ret_dict, header_name='Grouping instructions by Operation')
    utils.tabulate_stats(ext_ret_dict, header_name='Grouping instructions by Extension')
    ret_dict = instr_groups.privilege_modes(log,config)
    utils.tabulate_stats(ret_dict, header_name='Privilege Mode')

    if cycle_accurate_config != None:

        if 'cache' not in config['profiles']['cfg']['metrics'] or 'csr_compute' not in config['profiles']['cfg']['metrics']:
            logger.error("Cache and CSR compute metrics are not enabled. Please enable them for cycle accurate profiling.")
            raise SystemExit(1)

        for metric in config['profiles']['cfg']['metrics']:
            # Finding the new plugin file mentioned in the yaml file
            spec = importlib.util.spec_from_file_location("plugins", f"riscv_application_profiler/plugins/{metric}.py")
            # Converting file to a module
            metric_module = importlib.util.module_from_spec(spec)
            # Importing the module
            spec.loader.exec_module(metric_module)
            
            for funct in config['profiles']['cfg']['metrics'][metric]:
                funct_to_call = getattr(metric_module, funct)
                ret_dict1 = funct_to_call(master_inst_dict, ops_dict=op_dict, extension_used=extension_list, config= config, cycle_accurate_config=cycle_accurate_config)
                utils.tabulate_stats(ret_dict1, header_name=funct)

        # total_cycles = op_dict['total_cycles']
        total_cycles = sum([master_inst_dict[entry] for entry in master_inst_dict]) + cycle_accurate_config['cycles']['reset_cycles']
        ret_dict = {"Total Cycles": [total_cycles]}
        utils.tabulate_stats(ret_dict, header_name='Total Cycles')
        

