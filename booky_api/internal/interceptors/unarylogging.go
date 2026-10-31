package interceptors

import (
	"context"
	"log"

	"google.golang.org/grpc"
)

func UnaryLoggingInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (interface{}, error) {
	resp, err := handler(ctx, req)

	if err != nil {
		log.Printf("Method %s FAILED with error: %v", info.FullMethod, err)
	} else {
		log.Printf("Method %s SUCCESS", info.FullMethod)
	}

	return resp, err
}
