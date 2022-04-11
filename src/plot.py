from wds import WheatDetectionSystem

wds = WheatDetectionSystem('seedlings_2019', '06_03_test', 'frcnn', '400')
wds.read_metadata()

grid_size_m = 0.2
wds.create_map()
wds.draw_protocol()
wds.draw_vegetation([grid_size_m])
# wds.draw_images_on_map(False) # , tile_size_over = 1.0]
wds.draw_masks(grid_size_m)
wds.save_map()