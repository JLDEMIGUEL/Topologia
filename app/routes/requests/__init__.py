from marshmallow import Schema, fields


class FacesRequest(Schema):
    faces = fields.List(fields.List(fields.Field()), required=True, load_only=True)


class FaceAndFacesRequest(FacesRequest):
    face = fields.List(fields.Field(), required=True, load_only=True)


class PointsRequest(Schema):
    points = fields.List(fields.Field(), required=True, load_only=True)


class BasicAllRequest(Schema):
    n_faces_dim = fields.Int()
    skeleton_dim = fields.Int()
    boundary_matrix_dim = fields.Int()
    star_face = fields.List(fields.Field())
    closed_star_face = fields.List(fields.Field())
    link_face = fields.List(fields.Field())


class SimplicialAllRequest(FacesRequest, BasicAllRequest):
    pass


class ComplexAllRequest(PointsRequest, BasicAllRequest):
    pass
