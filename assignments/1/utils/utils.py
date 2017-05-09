def obj_keys(obj):
	return (k for k in obj.keys() if not k.startswith('__'))