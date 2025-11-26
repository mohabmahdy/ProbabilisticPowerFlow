import pandapower as pp
import math
import json
import numpy as np

def get_branches(net, z_base):
    all_branches = []
    for i in range(len(net.line)):
        line_i = net.line.loc[i]
        from_bus = line_i['from_bus']
        to_bus = line_i['to_bus']

        R = line_i['r_ohm_per_km'] * line_i['length_km']
        X = line_i['x_ohm_per_km'] * line_i['length_km']
        c_nf = line_i['c_nf_per_km'] * line_i['length_km']
        g_us = line_i['g_us_per_km'] * line_i['length_km']

        G = g_us * 1e-6
        B = 2*np.pi*50*c_nf*1e-9

        Z = complex(R, X)/z_base
        Z_s = complex(G, B)/z_base
        all_branches.append((from_bus, to_bus, Z, Z_s))
    return all_branches

def build_PecanNetwork():
    net = pp.create_empty_network(sn_mva=0.4)

    # --- BUSES
    for i in range(26):
        pp.create_bus(net, vn_kv=0.4, name=f"Bus {i}")

    # --- SOURCE
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, name="Grid Connection")

    # --- LINE DATA (from_bus, to_bus, length_km, std_type)
    # Main feeder: 0-1-2-...-10, 25
    # Laterals: from bus 2, 5, 8 → small branches
    line_data = [
        (0, 1, 0.15, "NAYY 4x150 SE"),
        (1, 2, 0.12, "NAYY 4x150 SE"),
        (2, 3, 0.10, "NAYY 4x150 SE"),
        (3, 4, 0.12, "NAYY 4x150 SE"),
        (4, 5, 0.10, "NAYY 4x150 SE"),
        (5, 6, 0.10, "NAYY 4x150 SE"),
        (6, 7, 0.12, "NAYY 4x150 SE"),
        (7, 8, 0.10, "NAYY 4x150 SE"),
        (8, 9, 0.09, "NAYY 4x150 SE"),
        (9, 10, 0.09, "NAYY 4x150 SE"),
        (10, 25, 0.09, "NAYY 4x150 SE"),

        # laterals
        (2, 11, 0.08, "NAYY 4x150 SE"),
        (11, 12, 0.08, "NAYY 4x150 SE"),
        (5, 13, 0.10, "NAYY 4x150 SE"),
        (13, 14, 0.10, "NAYY 4x150 SE"),
        (8, 15, 0.12, "NAYY 4x150 SE"),
        (15, 16, 0.12, "NAYY 4x150 SE"),
        (16, 17, 0.08, "NAYY 4x150 SE"),
        (4, 18, 0.10, "NAYY 4x150 SE"),
        (18, 19, 0.10, "NAYY 4x150 SE"),
        (7, 20, 0.12, "NAYY 4x150 SE"),
        (20, 21, 0.10, "NAYY 4x150 SE"),
        (21, 22, 0.09, "NAYY 4x150 SE"),
        (9, 23, 0.08, "NAYY 4x150 SE"),
        (10, 24, 0.08, "NAYY 4x150 SE"),
    ]

    for fb, tb, l, std in line_data:
        pp.create_line(net, from_bus=fb, to_bus=tb, length_km=l, std_type=std,
                       name=f"Line {fb}-{tb}")

    # --- LOADS (on all buses except 0)
    # kW loads (pf=0.95 lag)
    p_kw = {
        1: 20, 2: 30, 3: 25, 4: 15, 5: 35, 6: 18, 7: 20, 8: 25, 9: 1, 10: 2,
        11: 5, 12: 8, 13: 16, 14: 12, 15: 14, 16: 10, 17: 8, 18: 10, 19: 12,
        20: 15, 21: 10, 22: 8, 23: 1, 24: 3, 25:3
    }
    pf = 0.98

    for bus, p in p_kw.items():
        q = p * math.tan(math.acos(pf))
        pp.create_load(net, bus=bus, p_mw=p/1000, q_mvar=q/1000, name=f"Load {bus}")

    # set c to zero
    for i in range(len(net.line)):
        net.line.loc[i, "c_nf_per_km"] = 0.0
        net.line.loc[i, "g_us_per_km"] = 0.0

    # --- GEODATA (simple radial tree layout)
    x_pos = {}
    y_pos = {}
    # Main feeder along x
    for i in range(11):
        x_pos[i] = i*100.0
        y_pos[i] = 0.0
    # Laterals placed vertically
    lateral_coords = {
        11: (x_pos[2], -100), 12: (x_pos[2], -200),
        13: (x_pos[5], 100), 14: (x_pos[5], 200),
        15: (x_pos[8], -100), 16: (x_pos[8], -200), 17: (x_pos[8], -300),
        18: (x_pos[4], -100), 19: (x_pos[4], -200),
        20: (x_pos[7], 100), 21: (x_pos[7], 200), 22: (x_pos[7], 300),
        23: (x_pos[9], 100),
        24: (x_pos[10], -100), 25: (x_pos[10]+100, 0)
    }
    coords = {**{i: (x_pos[i], y_pos[i]) for i in range(11)}, **lateral_coords}

    for bus, (xx, yy) in coords.items():
        net.bus.at[bus, "geo"] = json.dumps({
            "type": "Point", "coordinates": [xx, yy]
        })

    return net

def build_new_pecan_network():
    net = pp.pandapower.networks.dickert_lv_networks.create_dickert_lv_network(feeders_range='middle', linetype='cable', customer='multiple', case='good', trafo_type_name='0.4 MVA 20/0.4 kV', trafo_type_data=None)

    pp.drop_trafos(net, [0])
    net.ext_grid.loc[0,"bus"]=1
    pp.drop_buses(net, [0])


    ## adding 5 new loades on 5 new busses
    n_bus_to_add = 5
    bus_to_attach = [26,31,40,42,46]
    new_bus = [None]*n_bus_to_add
    for i in range(n_bus_to_add):
        new_bus[i] = pp.create_bus(net, vn_kv=0.4)
        pp.create_load(net, bus=new_bus[i], p_mw=0, q_mvar=0)
        pp.create_line(net, from_bus=bus_to_attach[i], to_bus=new_bus[i], length_km=0.04, std_type="NAYY 4x150 SE")

    net.bus.reset_index()
    net.trafo.reset_index()
    net.line.reset_index()

    for i in net.line.index:
        net.line.loc[i, "c_nf_per_km"] = 0.0
        net.line.loc[i, "g_us_per_km"] = 0.0

    return net