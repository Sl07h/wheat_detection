from wds import WheatDetectionSystem

wds = WheatDetectionSystem('Field2_3_2019', '07_25', 'frcnn', '512')
# wds = WheatDetectionSystem('Field_2021', '1', 'frcnn', '512')
wds.draw_wheat_plots()
wds.draw_protocol()
wds.draw_grids([0.25, 0.5])
wds.save_map()
