from marshmallow import Schema, fields


class ThresholdValuesResponse(Schema):
    threshold_values = fields.List(fields.Field())


class FacesResponse(Schema):
    faces = fields.List(fields.List(fields.Field()))


class FacesDictResponse(Schema):
    faces = fields.Dict()


class BasicAllResponse(Schema):
    faces_list = fields.List(fields.List(fields.Field()))
    dimension = fields.Int()
    euler = fields.Int()
    components = fields.Int()
    bettis = fields.List(fields.Int)
    general_boundary_matrix = fields.List(fields.List(fields.Int))
    n_faces = fields.List(fields.List(fields.Field()))
    skeleton = fields.List(fields.List(fields.Field()))
    boundary_matrix = fields.List(fields.List(fields.Int))
    star = fields.List(fields.List(fields.Field()))
    closed_star = fields.List(fields.List(fields.Field()))
    link = fields.List(fields.List(fields.Field()))


class ComplexAllResponse(BasicAllResponse, FacesResponse, ThresholdValuesResponse):
    filtration_order = fields.List(fields.List(fields.Field()))
