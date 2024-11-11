import torch


def gradient_t(v, dt):
    '''
    Compute the gradient of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, t, ...)
    :return: 2D array of shape (b, h, w, t, ...)
    Compute the boundary gradient using the edge_order=1 option.
    '''

    # Get the shape of the input array
    b, h, w = v.shape[0], v.shape[1], v.shape[2]
    assert h == w

    # Get the shape of the input array
    assert len(v.shape) >= 4
    # Compute the derivatives with respect to y (vertical, height)
    grad_t = torch.gradient(v, dim=3, edge_order=1, spacing=dt)[0]

    return grad_t


def gradient_xy_vector(v, dx, dy):
    '''
    Compute the gradient of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, ..., 2)
    :return: 2D array of shape (b, h, w, ..., 2)
    Compute the boundary gradient using the edge_order=1 option.
    '''

    # Get the shape of the input array
    b, h, w = v.shape[0], v.shape[1], v.shape[2]
    _ = v.shape[-1]
    assert h == w
    assert _ == 2

    # Separate the components of v along the height (y-direction) and width (x-direction)
    vy, vx = v[..., 0], v[..., 1]

    # Compute the gradient of v_y with respect to the y-direction (height)
    grad_vy_y = torch.gradient(vy, dim=1, edge_order=1, spacing=dy)[0]  # d(vy)/dy

    # Compute the gradient of v_x with respect to the x-direction (width)
    grad_vx_x = torch.gradient(vx, dim=2, edge_order=1, spacing=dx)[0]  # d(vx)/dx

    # Combine the gradients into a single tensor of shape (H, W, 2)
    grad_v = torch.stack([grad_vy_y, grad_vx_x], dim=-1)

    return grad_v


def gradient_xy_scalar(v, dx, dy):
    '''
    Compute the gradient of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, ..., 1)
    :return: 2D array of shape (b, h, w, ..., 1)
    Compute the boundary gradient using the edge_order=1 option.
    '''

    # Get the shape of the input array
    b, h, w = v.shape[0], v.shape[1], v.shape[2]
    _ = v.shape[-1]
    assert h == w
    assert _ == 1

    v = v[..., 0]

    # Compute the derivatives with respect to y (vertical, height, 1st dimension, y)
    grad_y = torch.gradient(v, dim=1, edge_order=1, spacing=dy)[0]

    # Compute the derivatives with respect to x (horizontal, width, 2nd dimension, x)
    grad_x = torch.gradient(v, dim=2, edge_order=1, spacing=dx)[0]

    # Combine the gradients into a single tensor with shape (h, w, ..., 2)
    grad_v = torch.stack((grad_y, grad_x), dim=-1)

    return grad_v


def gradient_xx_scalar(v, dx):
    '''
    Compute the gradient of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, ..., 1)
    :return: 2D array of shape (b, h, w, ..., 1)
    Compute the boundary gradient using the edge_order=1 option.
    '''

    # Get the shape of the input array
    b, h, w = v.shape[0], v.shape[1], v.shape[2]
    _ = v.shape[-1]
    assert h == w
    assert _ == 1

    v = v[..., 0]

    # Compute the derivatives with respect to x (horizontal, width, 2nd dimension, x)
    grad_x = torch.gradient(v, dim=2, edge_order=1, spacing=dx)[0]

    # Compute the derivatives with respect to x (horizontal, width, 2nd dimension, x)
    grad_grad_x = torch.gradient(grad_x, dim=2, edge_order=1, spacing=dx)[0]

    return grad_grad_x.unsqueeze(-1)


def gradient_yy_scalar(v, dy):
    '''
    Compute the gradient of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, ..., 1)
    :return: 2D array of shape (b, h, w, ..., 1)
    Compute the boundary gradient using the edge_order=1 option.
    '''

    # Get the shape of the input array
    b, h, w = v.shape[0], v.shape[1], v.shape[2]
    _ = v.shape[-1]
    assert h == w
    assert _ == 1

    v = v[..., 0]

    # Compute the derivatives with respect to y (vertical, height, 1st dimension, y)
    grad_y = torch.gradient(v, dim=1, edge_order=1, spacing=dy)[0]

    # Compute the derivatives with respect to x (horizontal, width, 2nd dimension, x)
    grad_grad_y = torch.gradient(grad_y, dim=1, edge_order=1, spacing=dy)[0]

    return grad_grad_y.unsqueeze(-1)


def gradient_FDM_vector(v, dx, dy):
    '''
    Compute the gradient of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, ..., 2)
    :return: 2D array of shape (b, h, w, ..., 2)
    Set the boundary to zero.
    '''

    # Get the shape of the input array
    b, h, w = v.shape[0], v.shape[1], v.shape[2]
    _ = v.shape[-1]
    assert h == w
    assert _ == 2

    # Initialize the gradient array
    grad_v = torch.zeros_like(v, device=v.device)

    # Compute the derivatives with respect to y
    grad_v[:, 1:-1, :, 0] = (v[:, 2:, :, 0] - v[:, :-2, :, 0]) / (2 * dy)

    # Compute the derivatives with respect to x
    grad_v[:, :, 1:-1, 1] = (v[:, :, 2:, 1] - v[:, :, :-2, 1]) / (2 * dx)

    return grad_v


def laplacian(v, dx, dy):
    '''
    Compute the Laplacian of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, 2)
    :return: 2D array of shape (b, h, w, 2)
    Compute the boundary gradient using the edge_order=1 option.
    '''

    # Compute the gradient of the input tensor v
    grad_v = gradient_xy_vector(v, dx, dy)
    # Math
    # grad_v[..., 0] = d(vy)/dy
    # grad_v[..., 1] = d(vx)/dx

    # Compute the unmixed second derivatives
    grad_vy_yy = torch.gradient(grad_v[..., 0], dim=1, spacing=dy)[0]  # d^2(vy)/dy^2
    grad_vy_xx = torch.gradient(grad_v[..., 1], dim=2, spacing=dx)[0]  # d^2(vx)/dx^2

    # Sum the second derivatives to obtain the Laplacian
    laplacian_v = grad_vy_yy + grad_vy_xx  # d^2(vy)/dy^2 + d^2(vx)/dx^2

    return laplacian_v


if __name__ == '__main__':
    import torch

    is_M1 = False
    is_M2 = True
    is_M3 = True
    is_T1 = True
    is_Lap = True


    dx = 0.1
    dy = 0.1
    dt = 0.1

    # set the seed
    seed = 42
    torch.manual_seed(seed)

    # create a 2D array in torch, randomize it
    u = torch.rand(4, 10, 10, 2, requires_grad=False)  # Bs, H, W, 2
    print(u.shape)

    # M2
    if is_M2:
        grad_u_xy_m2 = gradient_xy_vector(u, dx, dy)
        print(grad_u_xy_m2.shape)

    # M3
    if is_M3:
        grad_u_xy_m3 = gradient_FDM_vector(u, dx, dy)
        print(grad_u_xy_m3.shape)

    # T1
    if is_T1:
        grad_u_t_t1 = gradient_t(u, dt)
        print(grad_u_t_t1.shape)

    # Lap
    if is_Lap:
        lap_u = laplacian(u, dx, dy)
        print(lap_u.shape)

    # Compare the results
    if is_M2 and is_M3:
        err_23 = torch.abs(grad_u_xy_m2 - grad_u_xy_m3).max()
        print('AbsErr M2-M3: {}'.format(err_23))
        err_cen_23 = torch.abs(grad_u_xy_m2[:, 1:-1, 1:-1, ...] - grad_u_xy_m3[:, 1:-1, 1:-1, ...]).max()
        print('AbsErr M2-M3 (center): {}'.format(err_cen_23))


