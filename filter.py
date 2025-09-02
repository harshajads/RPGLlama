from Level_generator import is_playable, validate_tilemap, get_promt_enemy_barrier_count, get_total_counts, validate_tilemap_with_level

def validate_tilemap(tilemap, level=None):
    if level is not None:
        return validate_tilemap_with_level(tilemap, level)
    else:
        bool_list, _ = validate_tilemap(tilemap)
        return all(bool_list)

__all__ = ['is_playable', 'validate_tilemap', 'get_promt_enemy_barrier_count', 'get_total_counts']