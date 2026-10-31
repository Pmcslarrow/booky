
.PHONY: protobufs
protobufs:
	protoc -I . -I proto \
		--go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		--grpc-gateway_out=. --grpc-gateway_opt=paths=source_relative \
		proto/booky.proto

reset:
	rm -rf vendor
	make protobufs
	go mod tidy
	go mod vendor

run: 
	go run cmd/main.go
