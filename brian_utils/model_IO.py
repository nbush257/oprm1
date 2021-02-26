from brian2 import *
import yaml
import pandas as pd

def import_eqs_from_txt(txt_file):
    '''
    Utility function to read in a brian2 model from a text file
    :param txt_file:
    :return:
    '''
    with open(txt_file,'r') as fid:
        eqs = fid.readlines()
        eqs = ''.join(eqs)
    return(eqs)


def import_eqs_from_yaml(yaml_file):
    '''
    Parse equations from a yaml file
    :param yaml_file:
    :return:
    '''
    with open(yaml_file,'r') as fid:
        eqs = yaml.load(fid,yaml.UnsafeLoader)
    return(eqs)


def import_namespace_from_yaml(yaml_file):
    '''
    Read parameters in a yaml file and parse as a dict
    to be sent to Brian2 groups.
    Parses all values as strings, then runs eval.
    Probably very unsafe to injections.

    :param yaml_file:
    :return: ns - dict of namespace variables
    '''
    with open(yaml_file,'r') as fid:
        ns_raw = yaml.load(fid,yaml.UnsafeLoader)
    ns = {}
    for k,v in ns_raw.items():
        ns[k] = eval(str(v))
    return(ns)


def export_ns_to_yaml(ns,yaml_file):
    #TODO: verify with arbitrary namespace
    '''
    Writes a namespace dict to a yaml file.
    Allows writing of brian2 unit types as strings
    :param ns: dictionary of parameters to write
    :param yaml_file: Path of file to write to
    :return:
    '''
    ns_str = {}
    for k,v in ns.items():
        if type(v) is not units.fundamentalunits.Quantity:
            ns_str[k] = v
        else:
            print(k)
            unit_str = v.get_best_unit().__repr__()
            # Prevents roundtrip failure for some special units (e.g. fkatal?)
            try:
                eval(unit_str)
            except NameError:
                unit_str = v.get_best_unit().dimensions.__repr__()
                value_str = str(v.base)
                ns_str[k] = '*'.join([value_str, unit_str])
                continue
            value_str = str(v/v.get_best_unit())
            ns_str[k] = '*'.join([value_str,unit_str])
    with open(yaml_file,'w') as fid:
        yaml.dump(ns_str,fid)
    return(0)


def gen_excitatory_destexhe():
    ''' Excitatory Destexhe synapse (1994)'''
    syn_eqs = '''
        ds/dt = ((1-s) * msyninf - s)/tau_syn_e : 1 (clock-driven)
        msyninf = 1 / (1+exp((v_pre - Qs)/ssyn)) : 1
        g_syne_post = gsyne_max(t)*s : siemens (summed)
        '''
    return(syn_eqs)


def gen_inhibitory_destexhe():
    ''' Inhibitory Destexhe synapse (1994)'''
    syn_eqs = '''
        ds/dt = ((1-s) * msyninf - s)/tau_syn_i : 1 (clock-driven)
        msyninf = 1 / (1+exp((v_pre - Qs)/ssyn)) : 1
        g_syni_post = gsyni_max(t)*s : siemens (summed)
        '''
    return(syn_eqs)


def gen_synapses(EI,model='destexhe'):
    '''
    Generic function to generate synapse equations
    :param EI: string ('e','i') to chose excitatory or inhibitory synapses
    :param model: class of model. Currently only destexhe is implemented
    :return:
    '''
    is_implemented = ['destexhe']
    if model!='destexhe':
        raise NotImplemented(f'Declared model type {model} is not implemented. Current model types are {is_implemented}')
    if EI.lower()[0]=='e':
        if model == 'destexhe':
            return(gen_excitatory_destexhe())
    if EI.lower()[0]=='i':
        if model == 'destexhe':
            return(gen_inhibitory_destexhe())


def import_edgelist(csv_file):
    '''
    Parses a csv file edgelist for passing to brian2
    :param csv_file:
    :return: presynaptic,postsynaptic,N
    '''
    df = pd.read_csv(csv_file)
    presynaptic = df['i'].values
    postsynaptic = df['j'].values
    N = np.max(df.values)+1

    return(presynaptic,postsynaptic,N)




