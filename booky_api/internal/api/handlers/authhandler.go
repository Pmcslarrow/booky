package handlers

import (
	booky "bookyapi/proto"
	"context"
)

type AuthHandler interface {
	Login(ctx context.Context, req *booky.LoginRequest) (*booky.LoginResponse, error)
}

type authHandler struct {
	booky.UnimplementedAuthServiceServer
}

func NewAuthHandler() *authHandler {
	return &authHandler{}
}

func (*authHandler) Login(ctx context.Context, req *booky.LoginRequest) (*booky.LoginResponse, error) {
	return nil, nil // TODO
}
