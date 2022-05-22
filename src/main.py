from wds import WheatDetectionSystem

wds = WheatDetectionSystem('Field2_3_2019', '07_25')

wds.create_map()
wds.draw_protocol()
wds.draw_wheat_plots('frcnn', 512)
wds.draw_wheat_grid('frcnn', 512)
wds.draw_tiles(['src', 'tgi'])
wds.draw_vegetation_cover(['tgi'])
wds.save_map()
