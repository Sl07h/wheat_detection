from wds import WheatDetectionSystem

# wds = WheatDetectionSystem('Field2_3_2019', '07_25', 'frcnn', '512')
wds = WheatDetectionSystem('2021_07_29_wheat_east', '07_29', 'frcnn', '400')
wds.draw_images_on_map(True)
# wds.draw_wheat_plots()
# wds.draw_protocol()
# wds.draw_grids([0.25, 0.5])
wds.save_map()
